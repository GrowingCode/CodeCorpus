package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;

public class AddProducer {

	public static void main(String[] args) {
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream(
					"C:\\添加PDF文档制作者.pdf"));// 关联文档对象与输出流
			document.addProducer();// 向文档中添加制作者
			document.open();// 打开文档
			document.add(new Paragraph("Add Producer."));// 向文档中添加内容
			document.close();// 关闭文档
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	}
}
