package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfCopy;
import com.itextpdf.text.pdf.PdfReader;

public class SplitPDFDocument {
	public static void main(String[] args) {
		String filePathFile = "c:\\原文档.pdf ";// 需要拆分的原文档
		PdfReader reader = null;// 声明PdfReader对象
		try {
			reader = new PdfReader(filePathFile);// 创建PdfReader对象
		} catch (IOException e) {
			e.printStackTrace();
		}
		int pageN = reader.getNumberOfPages();// 获取文件内的页数
		for (int i = 0; i < pageN; i++) {// 循环向外拆分页
			Document document = new Document(reader
					.getPageSizeWithRotation(i + 1));// 创建文档 同时获得前面循环的页
			PdfCopy copy = null;
			try {
				int len = filePathFile.length();// 获得文件完整路径的长度
				String noExt = filePathFile.substring(0, len - 5);// 去除文件扩展名后的路径
				String fileName = noExt + "-" + (i + 1) + ".pdf";// 拆分后生成的文件名称
				copy = new PdfCopy(document, new FileOutputStream(fileName));// 创建拷贝并关联文档与输出流对象
				document.open();// 打开文档
				copy.addPage(copy.getImportedPage(reader, i + 1));// 根据获得的页创建新文档
				document.close();// 关闭文档
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (DocumentException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
}
