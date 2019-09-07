package com.zzk;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import org.apache.pdfbox.pdfparser.PDFParser;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.util.PDFTextStripper;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.pdf.PdfReader;
import com.itextpdf.text.pdf.PdfStamper;

public class ReadPwdPdfFile {

	public static void main(String[] args) throws MalformedURLException {
		try {
			PdfReader reader = new PdfReader("c:\\设置密码.pdf", "123".getBytes());// 创建“水印.pdf”的PdfReader对象
			PdfStamper stamp = new PdfStamper(reader, new FileOutputStream(
					"c:\\TempFile.pdf"));// 创建PdfStamper对象
			stamp.close();// 关闭PdfStamper对象，并将从“设置密码.pdf”中读取的内容写入“TempFile.pdf”
			Document document = new Document();// 创建文档对象
			File file = new File("c:\\TempFile.pdf");// 创建File对象
			try {
				FileInputStream in = new FileInputStream(file);// 创建输入流对象
				PDFParser parser = new PDFParser(in);// 创建PDF
				parser.parse();// 解析PDF文档
				PDDocument pdfdocument = parser.getPDDocument();// 获得解析后的PDF文档
				PDFTextStripper stripper = new PDFTextStripper();// 创建PDF文本剥离器
				String msg = stripper.getText(pdfdocument);// 使用剥离器从PDF文档中剥离文本信息
				System.out.println("请取到加密的PDF文本信息如下:\n" + msg);// 输出信息
				in.close();// 关闭输入流对象
			} catch (Exception e) {
				e.printStackTrace();
			}
			document.close();// 关闭文档
			file.delete();// 删除“TempFile.pdf”
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
