package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;

public class SetPassWord {
	
	
	public static void main(String[] args) {
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter writer = PdfWriter.getInstance(document,
					new FileOutputStream("C:\\设置密码.pdf"));// 关联文档对象与输出流
			writer.setEncryption("zzk".getBytes(), "123".getBytes(),
					PdfWriter.ALLOW_COPY, PdfWriter.STANDARD_ENCRYPTION_128);// 设置密码参数和常量
			document.open();// 打开文档
			document.add(new Paragraph("Set Encryption"));// 向文档添加内容
			document.close();// 关闭文档
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	}
}
